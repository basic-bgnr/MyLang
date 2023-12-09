use std::{collections::HashMap, format, io::Write, println, unimplemented, unreachable, vec};

fn main() {
    Interpreter::new().interactive_prompt();
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum KeywordToken {
    Let,
    True,
    False,
    While,
}
impl KeywordToken {
    fn get_all() -> [KeywordToken; 4] {
        [Self::Let, Self::True, Self::False, Self::While]
    }
    fn value(&self) -> &str {
        match self {
            Self::Let => "let",
            Self::True => "true",
            Self::False => "false",
            Self::While => "while",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum SymbolToken {
    SmallBracketOpen,
    SmallBracketClose,
    Dot,
    Equal,
    UnderScore,
    SemiColon,
    CurlyBracketOpen,
    CurlyBracketClose,
}
impl SymbolToken {
    fn get_all() -> [SymbolToken; 8] {
        [
            Self::SmallBracketOpen,
            Self::SmallBracketClose,
            Self::Dot,
            Self::Equal,
            Self::UnderScore,
            Self::SemiColon,
            Self::CurlyBracketOpen,
            Self::CurlyBracketClose,
        ]
    }
    fn value(&self) -> &str {
        match self {
            Self::SmallBracketOpen => "(",
            Self::SmallBracketClose => ")",
            Self::Dot => ".",
            Self::Equal => "=",
            Self::UnderScore => "_",
            Self::SemiColon => ";",
            Self::CurlyBracketOpen => "{",
            Self::CurlyBracketClose => "}",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}
#[derive(PartialEq, Debug, Clone, Copy)]
enum OperatorToken {
    PLUS,
    MINUS,
    STAR,
    DIVIDE,

    LOGICAL_AND,
    LOGICAL_OR,

    LOGICAL_NOT,

    LESS_THAN,
    GREATER_THAN,
    EQUAL_TO,
}
impl OperatorToken {
    fn get_all() -> [OperatorToken; 10] {
        [
            Self::PLUS,
            Self::MINUS,
            Self::STAR,
            Self::DIVIDE,
            Self::LOGICAL_AND,
            Self::LOGICAL_OR,
            Self::LOGICAL_NOT,
            Self::LESS_THAN,
            Self::GREATER_THAN,
            Self::EQUAL_TO,
        ]
    }
    fn value(&self) -> &str {
        match self {
            Self::PLUS => "+",
            Self::MINUS => "-",
            Self::STAR => "*",
            Self::DIVIDE => "/",
            Self::LOGICAL_AND => "&&",
            Self::LOGICAL_OR => "||",
            Self::LOGICAL_NOT => "!",
            Self::LESS_THAN => "<",
            Self::GREATER_THAN => ">",
            Self::EQUAL_TO => "==",
        }
    }
    //output type after application of operator
    fn tipe(&self) -> &LanguageType {
        match self {
            Self::PLUS | Self::MINUS | Self::STAR | Self::DIVIDE => &LanguageType::Number,
            Self::LOGICAL_AND
            | Self::LOGICAL_OR
            | Self::LOGICAL_NOT
            | Self::LESS_THAN
            | Self::GREATER_THAN
            | Self::EQUAL_TO => &LanguageType::Boolean,
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Token<'a> {
    Digit(&'a char, TokenInfo),
    Alphabet(&'a char, TokenInfo),
    Identifier(&'a [char], TokenInfo),

    Operator(OperatorToken, TokenInfo),
    Symbol(SymbolToken, TokenInfo),

    Keyword(KeywordToken, TokenInfo),

    WhiteSpace(TokenInfo),

    EOF(TokenInfo),

    Alphanumeric(&'a [char]),
}
impl<'a> Token<'a> {
    fn get_token_info(&self) -> &TokenInfo {
        match self {
            Self::Digit(_, token_info) => token_info,
            Self::Alphabet(_, token_info) => token_info,
            Self::Identifier(_, token_info) => token_info,
            Self::Operator(_, token_info) => token_info,
            Self::Symbol(_, token_info) => token_info,
            Self::Keyword(_, token_info) => token_info,
            Self::EOF(token_info) => token_info,

            Self::WhiteSpace(token_info) => token_info,
            Self::Alphanumeric(_) => unreachable!(),
        }
    }
    fn value(&self) -> String {
        match self {
            Self::Digit(d, _) => d.to_string(),
            Self::Alphabet(a, _) => a.to_string(),
            Self::Identifier(i, _) => i.iter().collect::<String>(),
            Self::Operator(o, _) => o.value().to_string(),
            Self::Symbol(s, _) => s.value().to_string(),
            Self::Keyword(k, _) => k.value().to_string(),

            Self::WhiteSpace(_) => "".to_string(),
            Self::Alphanumeric(lst_chr) => lst_chr.iter().collect(),
            Self::EOF(_) => "<EOF>".to_string(),
        }
    }
}
#[derive(PartialEq, Debug, Clone, Copy)]
struct TokenInfo {
    line_number: usize,
    column_number: usize,
}
struct Tokenizer<'a> {
    input_string: &'a [char],
    index: usize,
    len: usize,
    column_number: usize,
    line_number: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input_string: &'a [char]) -> Self {
        Self {
            input_string: input_string,
            index: 0,
            len: input_string.len(),
            column_number: 1,
            line_number: 1,
        }
    }
    fn get_token_info(&self) -> TokenInfo {
        TokenInfo {
            line_number: self.line_number,
            column_number: self.column_number,
        }
    }

    fn advance(&mut self) {
        self.index += 1;
        self.column_number += 1;
    }

    fn peek(&self) -> Option<&char> {
        if self.index < self.len {
            Some(&self.input_string[self.index])
        } else {
            None
        }
    }

    fn reset_index_at(&mut self, new_index: usize) {
        let index_diff = self.index - new_index;
        self.index = new_index;
        self.column_number -= index_diff;
    }

    fn get_alphanumeric_token(&self, start_index: usize, end_index: usize) -> Option<Token<'a>> {
        if start_index >= end_index {
            None
        } else {
            Some(Token::Alphanumeric(
                &self.input_string[start_index..end_index],
            ))
        }
    }
    fn match_any_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial = self.index;
        match self.peek() {
            Some(&chr) if blob.contains(chr) => {
                self.advance();
                return self.get_alphanumeric_token(initial, self.index);
            }
            _ => {
                return None;
            }
        }
    }
    fn match_many_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial_index = self.index;
        loop {
            match self.peek() {
                Some(&chr) if blob.contains(chr) => self.advance(),
                _ => break,
            }
        }
        self.get_alphanumeric_token(initial_index, self.index)
    }
    fn match_exact_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial_index = self.index;
        for c in blob.chars() {
            match self.peek() {
                Some(&chr) if chr == c => self.advance(),
                _ => {
                    self.reset_index_at(initial_index);
                    return None;
                }
            }
        }
        self.get_alphanumeric_token(initial_index, self.index)
    }

    fn match_operator(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        for operator_token in OperatorToken::get_all() {
            match self.match_exact_in_blob(operator_token.value()) {
                Some(Token::Alphanumeric(a)) if operator_token.is_equal(a) => {
                    return Some(Token::Operator(operator_token, token_info));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_symbol(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        for symbol_token in SymbolToken::get_all() {
            match self.match_exact_in_blob(symbol_token.value()) {
                Some(Token::Alphanumeric(a)) if symbol_token.is_equal(a) => {
                    return Some(Token::Symbol(symbol_token, token_info));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_whitespace(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_many_in_blob(" ") {
            Some(_) => Some(Token::WhiteSpace(token_info)),
            _ => None,
        }
    }

    fn match_digit(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_any_in_blob("0123456789") {
            Some(Token::Alphanumeric(chr)) => Some(Token::Digit(chr.first().unwrap(), token_info)),
            _ => None,
        }
    }
    fn match_alphabet(&mut self) -> Option<Token<'a>> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_any_in_blob(&blob) {
            Some(Token::Alphanumeric(chr)) => {
                Some(Token::Alphabet(chr.first().unwrap(), token_info))
            }
            _ => None,
        }
    }
    fn match_identifier(&mut self) -> Option<Token<'a>> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_many_in_blob(&blob) {
            Some(Token::Alphanumeric(chr)) => Some(Token::Identifier(chr, token_info)),
            _ => None,
        }
    }
    fn match_keyword(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        let initial_index = self.index;
        for keyword_token in KeywordToken::get_all() {
            match self.match_exact_in_blob(keyword_token.value()) {
                Some(Token::Alphanumeric(_)) => match self.match_alphabet() {
                    Some(_) => {
                        self.reset_index_at(initial_index);
                        return None;
                    }
                    None => {
                        return Some(Token::Keyword(keyword_token, token_info));
                    }
                },
                _ => {
                    continue;
                }
            }
        }
        None
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, Vec<String>> {
        let mut tokens = Vec::new();
        let mut tokenization_errors = vec![];
        while self.peek().is_some() {
            if let Some(token) = self.match_keyword() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_digit() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_identifier() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_operator() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_symbol() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_whitespace() {
                tokens.push(token);
                continue;
            }
            let token_info = self.get_token_info();
            tokenization_errors.push(format!(
                "Tokenizing Error: Unknown token at line {}, column {}",
                token_info.line_number, token_info.column_number
            ));
            self.advance();
        }
        if tokenization_errors.len() > 0 {
            Err(tokenization_errors)
        } else {
            let token_info = self.get_token_info();
            tokens.push(Token::EOF(token_info));
            // println!("{:?}", tokens);
            Result::Ok(tokens)
        }
    }
}

struct Parser<'b, 'a: 'b> {
    tokens: &'a [Token<'a>],
    index: usize,
    back_track_index: usize,
    len: usize,
    // program: Vec<Either>,
    block_position: usize,
    num_identifiers_in_block: Vec<usize>,
    identifiers_list: Option<Vec<Either>>, //usize is the block_position of the identifier
    // prev_program: Option<&'b [Either]>,
    previous_identifiers: Option<&'b [Either]>,
}

impl<'b, 'a> Parser<'b, 'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        Parser {
            tokens: tokens,
            index: 0,
            back_track_index: 0, 
            len: tokens.len(),
            // program: Vec::new(),
            block_position: 0,
            num_identifiers_in_block: vec![0],
            identifiers_list: None,
            // prev_program: None,
            previous_identifiers: None,
        }
    }
    fn increment_num_identifier_in_current_block(&mut self) {
        let block_postion = self.get_block_position();
        self.num_identifiers_in_block[block_postion] += 1;
    }

    fn get_block_position(&self) -> usize {
        self.block_position
    }
    fn increment_block_position(&mut self) {
        self.block_position += 1;
        self.num_identifiers_in_block.push(0)
    }
    fn decrement_block_position(&mut self) {
        let current_block_position = self.block_position;
        let num_identifiers_to_pop = self.num_identifiers_in_block[current_block_position];
        for _ in 0..num_identifiers_to_pop {
            self.identifiers_list.as_mut().unwrap().pop();
        }
        self.block_position -= 1;
        self.num_identifiers_in_block.pop();
    }

    fn add_reference_to_previous_identifiers(
        &mut self,
        previous_identifiers: Option<&'b [Either]>,
    ) {
        self.previous_identifiers = previous_identifiers;
    }

    fn remember_index(&mut self) {
        self.back_track_index = self.index;
    }

    fn back_track(&mut self){
        self.index = self.back_track_index;
    }

    fn reset_index_at(&mut self, index: usize) {
        self.index = index;
    }

    fn get_identifiers_list(self) -> Option<Vec<Either>> {
        self.identifiers_list
    }

    fn push_identifier(&mut self, identifier: &Either) {
        if self.identifiers_list.is_none() {
            self.identifiers_list = Some(vec![identifier.clone()]);
        } else {
            self.identifiers_list
                .as_mut()
                .unwrap()
                .push(identifier.clone());
        }

        self.increment_num_identifier_in_current_block();
    }

    fn match_identifier_in_list(
        &self,
        name: &str,
        identifiers_list: Option<&'b [Either]>,
    ) -> Option<Either> {
        match identifiers_list {
            Some(identifiers) => {
                for identifier in identifiers.iter().rev() {
                    if let identifier @ Either::Identifier(identifier_name, _) = identifier {
                        if name == identifier_name {
                            return Some(identifier.clone());
                        }
                    }
                }
                None
            }
            None => None,
        }
    }

    fn get_reference_to_identifier(&self, name: &str) -> Option<Either> {
        self.match_identifier_in_list(name, self.identifiers_list.as_deref())
            .or_else(|| self.match_identifier_in_list(name, self.previous_identifiers))
    }

    fn parse_next_statement(&mut self) -> Result<Either, Error> {
        self.parse_while_statement()
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_let_statement(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_assignment_statement(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                Error::ParseError(_) => self.parse_logical_term(),
                Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_) => Err(e),
            }).or_else(|e| match e {
                e @ (Error::ParseError(_) | Error::TypeError(_) | Error::IdentifierError(_) | Error::SyntaxError(_)) => Err(e),
            }).map(|(statement, _)| statement)
    }

    fn parse(&mut self) -> Result<Vec<Either>, Error> {
        let mut statements = Vec::new();
        loop {
            let parsed_statement = self.parse_next_statement()?;
            self.consume_optional_semicolon();
            self.consume_optional_whitespace();
            statements.push(parsed_statement);

            if let Some(Token::EOF(_)) = self.peek() {
                break;
            }
        }
        // println!("{:?}", statements);
        Ok(statements)
    }

    fn advance(&mut self) -> Option<&Token> {
        if self.index < self.len {
            let return_value = &self.tokens[self.index];
            self.index += 1;
            Some(return_value)
        } else {
            None
        }
    }

    fn peek(&self) -> Option<&Token<'a>> {
        if self.index < self.len {
            Some(&self.tokens[self.index])
        } else {
            None
        }
    }

    fn consume_optional_semicolon(&mut self) -> bool {
        if let Some(Token::Symbol(SymbolToken::SemiColon, _)) = self.peek() {
            self.advance();
            return true;
        }
        false
    }

    fn consume_optional_whitespace(&mut self) -> bool {
        if let Some(Token::WhiteSpace(_)) = self.peek() {
            self.advance();
            return true;
        }
        false
    }

    fn parse_while_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::Keyword(keyword, token_info_while)) if keyword == KeywordToken::While => {
                self.advance();
                self.consume_optional_whitespace();
                let (condition_expression, _) = self.parse_logical_term()?;
                self.consume_optional_whitespace();
                let (block_expression, _) = self.parse_logical_term()?;
                let tipe = block_expression.tipe().clone();
                Ok((
                    Either::WhileStatement(
                        Box::new(condition_expression),
                        Box::new(block_expression),
                        tipe,
                    ),
                    token_info_while,
                ))
            }
            Some(token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse while statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_let_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::Keyword(keyword, token_info))
                if keyword.value() == KeywordToken::Let.value() =>
            {
                self.advance();
                self.advance(); //consume whitespace
                match self.peek().cloned() {
                    Some(identifier @ Token::Identifier(_, token_info)) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::Symbol(symbol, _))
                                if symbol.value() == SymbolToken::Equal.value() =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                let (term, token_info) = self.parse_logical_term()?;

                                let tipe = term.tipe();

                                match tipe {
                                    LanguageType::Void => Err(Error::TypeError(format!(
                                        "Type Error: Void type found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    ))),
                                    _ => {
                                        self.push_identifier(&Either::Identifier(
                                            identifier.value(),
                                            *tipe,
                                        ));

                                        Ok((
                                            LetStatement::new(identifier.value(), term),
                                            token_info,
                                        ))
                                    }
                                }
                            }
                            Some(token) => Err(Error::SyntaxError(format!(
                                "Syntax Error: No equal found at line {}, column {}",
                                token.get_token_info().line_number,
                                token.get_token_info().column_number
                            ))),
                            None => Err(Error::SyntaxError(format!(
                                "Parsing Error: None value encountered at line {}, {}",
                                token_info.line_number, token_info.column_number
                            ))),
                        }
                    }
                    Some(token) => Err(Error::SyntaxError(format!(
                        "Syntax Error: No identifier found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    ))),
                    None => Err(Error::SyntaxError(format!(
                        "Syntax Error: None value found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    ))),
                }
            }
            Some(token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse let statement at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }
    fn parse_assignment_statement(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        self.remember_index();
        match self.peek().cloned() {
            Some(identifier_token @ Token::Identifier(_, token_info)) => {
                match self.get_reference_to_identifier(&identifier_token.value()) {
                    Some(identifier) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::Symbol(symbol, _))
                                if symbol.value() == SymbolToken::Equal.value() =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                let (term, token_info) = self.parse_logical_term()?;
                                if term.tipe() == identifier.tipe() {
                                    Ok((
                                        AssignmentStatement::new(identifier_token.value(), term),
                                        token_info,
                                    ))
                                } else {
                                    Err(Error::TypeError(format!(
                                        "Type Error: Type mismatched, found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    )))
                                }
                            }
                            Some(other_token) => {
                                self.back_track();
                                Err(Error::ParseError(format!(
                                "Parsing Error: Cannot parse assignment statement found at line {}, column {}",
                                other_token.get_token_info().line_number,
                                other_token.get_token_info().column_number
                            ))) 
                            }
                            None => Err(Error::ParseError(format!(
                                "Parsing Error: Cannot parse assignment statement found at line {}, column {}",
                                token_info.line_number,
                                token_info.column_number
                            ))),
                        }
                    }
                    None => Err(Error::IdentifierError(format!(
                        "Identifier Error: use of undeclared variable {:?} encountered at line {}, {}",
                        identifier_token.value(),
                        token_info.line_number,
                        token_info.column_number
                    ))),
                }
            }
            Some(other_token) => Err(Error::ParseError(format!(
                "Parsing Error: Cannot parse assignment statement at line {}, column {}",
                other_token.get_token_info().line_number,
                other_token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_logical_term(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_comparison_term()?;
        // self.consume_optional_whitespace();
        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::LOGICAL_AND | OperatorToken::LOGICAL_OR),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_comparison_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    (OperatorToken::LOGICAL_AND | OperatorToken::LOGICAL_OR),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }
    fn parse_comparison_term(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_term()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::LESS_THAN
                    | OperatorToken::GREATER_THAN
                    | OperatorToken::EQUAL_TO),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(operator_token @ OperatorToken::EQUAL_TO, token_info))
                    if first_expression.tipe() == &LanguageType::Boolean =>
                {
                    self.advance();
                    let (second_expression, token_info) = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    operator_token @ (OperatorToken::LESS_THAN | OperatorToken::GREATER_THAN),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean => {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => return Ok((first_expression, token_info)),
            }
        }
    }

    fn parse_term(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_prod()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::PLUS | OperatorToken::MINUS),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_prod()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    operator_token @ (OperatorToken::PLUS | OperatorToken::MINUS),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean => {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }

    fn parse_prod(&mut self) -> Result<(Either, TokenInfo), Error> {
        let (mut first_expression, token_info) = self.parse_unary()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::STAR | OperatorToken::DIVIDE),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_unary()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = BinaryExpression::new(
                            first_expression,
                            second_expression,
                            operator_token,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(Error::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::Operator(
                    operator_token @ (OperatorToken::STAR | OperatorToken::DIVIDE),
                    token_info,
                )) if first_expression.tipe() == &LanguageType::Boolean => {
                    return Err(Error::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }

                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }

    fn parse_unary(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Operator(
                operator @ (OperatorToken::PLUS | OperatorToken::MINUS),
                token_info,
            )) => {
                self.advance();
                let (expr, token_info) = self.parse_unary()?;
                let tipe = expr.tipe().clone();
                let unary_expr = UnaryExpression::new(expr, operator);
                if tipe == LanguageType::Number {
                    Ok((unary_expr, token_info))
                } else {
                    Err(Error::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::Operator(operator @ OperatorToken::LOGICAL_NOT, token_info)) => {
                self.advance();
                let (expr, token_info) = self.parse_unary()?;
                let tipe = expr.tipe().clone();
                let unary_expr = UnaryExpression::new(expr, operator);
                if tipe == LanguageType::Boolean {
                    Ok((unary_expr, token_info))
                } else {
                    Err(Error::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::WhiteSpace(_)) => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_bracket(),
        }
    }

    fn parse_bracket(&mut self) -> Result<(Either, TokenInfo), Error> {
        match self.peek().cloned() {
            Some(Token::Symbol(SymbolToken::SmallBracketOpen, token_info)) => {
                self.advance();
                let (val, token_info) = self.parse_logical_term()?;
                match self.peek().cloned() {
                    Some(Token::Symbol(SymbolToken::SmallBracketClose, _)) => {
                        self.advance();
                        Ok((val, token_info))
                    }
                    Some(Token::WhiteSpace(_)) => {
                        self.advance();
                        self.parse_bracket()
                    }
                    Some(token) => Err(Error::ParseError(format!(
                        "Parsing Error: No closing bracket found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    ))),
                    None => Err(Error::ParseError(format!(
                        "Parsing Error: None value encountered"
                    ))),
                }
            }
            _ => self.parse_literal(),
        }
    }

    fn parse_literal(&mut self) -> Result<(Either, TokenInfo), Error> {
        self.consume_optional_whitespace();
        // println!("debug parse_literal {:?}", self.peek().cloned());
        match self.peek().cloned() {
            Some(Token::Digit(first_digit, digit_token)) => {
                let mut digits = vec![first_digit];
                let mut num_of_dot = 0;
                let mut token_info = digit_token;
                self.advance();
                loop {
                    match self.peek().cloned() {
                        Some(Token::Symbol(SymbolToken::Dot, _)) if num_of_dot < 1 => {
                            self.advance();
                            num_of_dot += 1;
                            digits.push(&'.');
                        }
                        Some(Token::Symbol(SymbolToken::Dot, token_info_symbol))
                            if num_of_dot >= 1 =>
                        {
                            return Err(Error::SyntaxError(format!(
                "Syntax Error: Extra decimal(.) encountered at line {}, column {}",
                token_info_symbol.line_number,
                token_info_symbol.column_number ,
            )));
                        }
                        Some(Token::Digit(val, digit_token)) => {
                            self.advance();
                            token_info = digit_token;
                            digits.push(val);
                        }

                        _ => {
                            break;
                        }
                    }
                }
                Ok((
                    Either::Number(
                        digits
                            .into_iter()
                            .collect::<String>()
                            .parse::<f64>()
                            .unwrap(),
                    ),
                    digit_token,
                ))
            }
            Some(Token::Keyword(KeywordToken::True, token_info)) => {
                self.advance();
                Ok((Either::Bool(true), token_info))
            }
            Some(Token::Keyword(KeywordToken::False, token_info)) => {
                self.advance();
                Ok((Either::Bool(false), token_info))
            }
            Some(Token::Identifier(var_name, token_info)) => {
                // println!("debug in parse_literal {:?}", var_name);
                self.advance();
                let var_name = var_name.iter().collect::<String>();
                match self.get_reference_to_identifier(&var_name) {
                    Some(identifier) => Ok((identifier.clone(), token_info)),
                    None => Err(Error::IdentifierError(format!(
                "Identifier Error: No variable named '{}' found in scope at line {}, column {}",
                var_name,
                token_info.line_number,
                token_info.column_number,
            ))),
                }

            }
            Some(Token::Symbol(SymbolToken::CurlyBracketOpen, token_info)) => {
                // println!("debug curly: {:?}", "here");
                self.increment_block_position();

                self.advance();
                self.consume_optional_whitespace();
                let mut statements:Vec<Either> = Vec::new();
                let mut last_statement_semicolon = false;
                loop {
                    match self.peek().cloned(){
                        Some(Token::Symbol(SymbolToken::CurlyBracketClose, _)) => {
                            self.decrement_block_position();

                            self.advance();
                            self.consume_optional_semicolon();
                            match statements.last() {
                                Some(last_statement) if !last_statement_semicolon => {
                                    let tipe = last_statement.tipe().clone();
                                    return Ok( (Either::BlockStatement(statements, tipe), token_info ) );
                                }
                                Some(_) => {
                                    return Ok( (Either::BlockStatement(statements, LanguageType::Void), token_info) );
                                }
                                None => {
                                    return Ok( (Either::BlockStatement(statements, LanguageType::Void), token_info) );
                                }
                            }
                        }
                        Some(_) =>  {
                            let statement = self.parse_next_statement()?;
                            last_statement_semicolon = self.consume_optional_semicolon();
                            self.consume_optional_whitespace();
                            // println!("debug curly: {:?} {:?}", statement, self.peek());
                            statements.push(statement);
                        }
                        None => {
                            return Err(Error::TypeError(format!("None encountered")));
                        }
                    }
                }
            }

            Some(token) => Err(Error::SyntaxError(format!(
                "Syntax Error: Value other than literal (Number, Boolean...) '{}' encountered at line {}, column {}",
                token.value(),
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            None => unreachable!(), 
        }
    }
}
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum LanguageType {
    Number,
    Boolean,
    Void,
}
#[derive(Debug)]
enum Either {
    Number(f64),
    BinaryExpression(Box<BinaryExpression>),
    UnaryExpression(Box<UnaryExpression>),
    LetStatement(Box<LetStatement>),
    AssignmentStatement(Box<AssignmentStatement>),
    Bool(bool),
    Identifier(String, LanguageType),
    BlockStatement(Vec<Either>, LanguageType),
    WhileStatement(Box<Either>, Box<Either>, LanguageType),
}

impl Either {
    fn clone(&self) -> Self {
        match self {
            Self::Identifier(name, tipe) => Self::Identifier(name.clone(), tipe.clone()),
            _ => unimplemented!(),
        }
    }

    fn tipe(&self) -> &LanguageType {
        match self {
            Self::Number(_) => &LanguageType::Number,
            Self::Bool(_) => &LanguageType::Boolean,
            Self::BinaryExpression(boxed_binary_expression) => boxed_binary_expression.tipe(),
            Self::UnaryExpression(boxed_unary_expression) => boxed_unary_expression.tipe(),
            Self::LetStatement(boxed_let_statement) => boxed_let_statement.tipe(),
            Self::AssignmentStatement(boxed_assignment_statement) => {
                boxed_assignment_statement.tipe()
            }
            Self::Identifier(_, tipe) => tipe,
            Self::BlockStatement(_, tipe) => tipe,
            Self::WhileStatement(_, _, tipe) => tipe,
        }
    }
}

#[derive(Debug)]
enum Error {
    ParseError(String),
    TypeError(String),
    IdentifierError(String),
    SyntaxError(String),
}

#[derive(Debug)]
struct BinaryExpression {
    left: Either,
    right: Either,
    operator: OperatorToken,
}
impl BinaryExpression {
    fn new(left: Either, right: Either, operator: OperatorToken) -> Either {
        Either::BinaryExpression(Box::new(Self {
            left: left,
            right: right,
            operator: operator,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.operator.tipe()
    }
}
#[derive(Debug)]
struct UnaryExpression {
    val: Either,
    operator: OperatorToken,
}
impl UnaryExpression {
    fn new(val: Either, operator: OperatorToken) -> Either {
        Either::UnaryExpression(Box::new(Self {
            val: val,
            operator: operator,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.operator.tipe()
    }
}

#[derive(Debug)]
struct AssignmentStatement {
    lvalue: String,
    rvalue: Either,
}

impl AssignmentStatement {
    fn new(lvalue: String, rvalue: Either) -> Either {
        Either::AssignmentStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.rvalue.tipe()
    }
}
#[derive(Debug)]
struct LetStatement {
    lvalue: String,
    rvalue: Either,
}

impl LetStatement {
    fn new(lvalue: String, rvalue: Either) -> Either {
        Either::LetStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.rvalue.tipe()
    }
}
///////////////////////////////////////////////////////// Interpreter Code /////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
enum InternalDataStucture {
    Number(f64),
    Bool(bool),
    Void,
}

impl InternalDataStucture {
    fn __match_binary_operation__(left: Self, right: Self, operator: OperatorToken) -> Self {
        match (left, right) {
            (Self::Number(l), Self::Number(r)) => match operator {
                OperatorToken::PLUS => Self::Number(l + r),
                OperatorToken::MINUS => Self::Number(l - r),
                OperatorToken::STAR => Self::Number(l * r),
                OperatorToken::DIVIDE => Self::Number(l / r),

                OperatorToken::LESS_THAN => Self::Bool(l < r),
                OperatorToken::GREATER_THAN => Self::Bool(l > r),
                OperatorToken::EQUAL_TO => Self::Bool(l == r),

                _ => {
                    println!("{:?}, {:?}", left, right);
                    unreachable!();
                }
            },

            (Self::Bool(l), Self::Bool(r)) => match operator {
                OperatorToken::LOGICAL_AND => Self::Bool(l && r),
                OperatorToken::LOGICAL_OR => Self::Bool(l || r),
                OperatorToken::EQUAL_TO => Self::Bool(l == r),
                _ => {
                    println!("{:?}, {:?}", left, right);
                    unreachable!()
                }
            },

            _ => {
                println!("{:?}, {:?}", left, right);
                unreachable!()
            }
        }
    }
    fn __match_unary_operation__(val: Self, operator: OperatorToken) -> Self {
        match val {
            Self::Number(v) => match operator {
                OperatorToken::PLUS => Self::Number(v),
                OperatorToken::MINUS => Self::Number(-v),
                _ => unreachable!(),
            },
            Self::Bool(b) => match operator {
                OperatorToken::LOGICAL_NOT => Self::Bool(!b),
                _ => unreachable!(),
            },

            _ => unreachable!(),
        }
    }
}
#[derive(Debug)]
struct Environment {
    container: Vec<HashMap<String, InternalDataStucture>>,
}

impl Environment {
    fn new() -> Self {
        Self {
            container: vec![HashMap::new()],
        }
    }
    fn insert_new(&mut self, lvalue: String, rvalue: InternalDataStucture) {
        self.container.last_mut().unwrap().insert(lvalue, rvalue);
    }
    fn insert(&mut self, lvalue: String, rvalue: InternalDataStucture) {
        let mut location = 0;
        for (i, child_container) in self.container.iter().enumerate().rev() {
            if child_container.contains_key(&lvalue) {
                location = i;
            }
        }
        self.container[location].insert(lvalue, rvalue);
    }
    fn get(&self, name: &str) -> Option<&InternalDataStucture> {
        // self.container.get(name).or_else(|| {
        //     self.parent
        //         .as_ref()
        //         .and_then(|parent_env| parent_env.get(name))
        // })
        for container in self.container.iter().rev() {
            if container.contains_key(name) {
                return container.get(name);
            }
        }
        None
    }
    fn create_child(&mut self) {
        self.container.push(HashMap::new());
    }
    fn pop_child(&mut self) {
        self.container.pop();
    }
}
struct Interpreter {
    environment: Environment,
    previous_identifiers: Option<Vec<Either>>,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            environment: Environment::new(),
            previous_identifiers: None,
        }
    }
    fn append_new_identifiers(&mut self, new_identifiers: Option<Vec<Either>>) {
        if let Some(mut new_identifiers) = new_identifiers {
            match self.previous_identifiers {
                Some(ref mut previous_identifiers) => {
                    previous_identifiers.append(&mut new_identifiers)
                }
                None => {
                    self.previous_identifiers = Some(new_identifiers);
                }
            }
        }
    }

    fn tokenize_and_parse(&self, input: &str) -> Result<(Vec<Either>, Option<Vec<Either>>), Error> {
        let input_chars = input.chars().collect::<Vec<_>>();

        let mut tokenizer = Tokenizer::new(&input_chars);
        let tokens = tokenizer.tokenize();

        match tokens {
            Ok(token_list) => {
                let mut parser = Parser::new(&token_list);
                parser.add_reference_to_previous_identifiers(self.previous_identifiers.as_deref());

                let parsed_statement = parser.parse()?;

                Ok((parsed_statement, parser.get_identifiers_list()))
            }
            Err(err) => Err(Error::ParseError(
                err.into_iter()
                    .map(|x| x + "\n")
                    .collect::<String>()
                    .trim_end()
                    .to_string(),
            )),
        }
    }
    fn calculate_statement(&mut self, statement: &Either) -> InternalDataStucture {
        match statement {
            Either::Number(val) => InternalDataStucture::Number(*val),

            Either::Bool(val) => InternalDataStucture::Bool(*val),

            Either::BinaryExpression(boxed_expression) => match &**boxed_expression {
                BinaryExpression {
                    left,
                    right,
                    operator,
                } => InternalDataStucture::__match_binary_operation__(
                    self.calculate_statement(&left),
                    self.calculate_statement(&right),
                    *operator,
                ),
            },
            Either::UnaryExpression(boxed_expression) => match &**boxed_expression {
                UnaryExpression { val, operator } => {
                    InternalDataStucture::__match_unary_operation__(
                        self.calculate_statement(&val),
                        *operator,
                    )
                }
            },
            Either::LetStatement(boxed_let_statement) => match &**boxed_let_statement {
                LetStatement { lvalue, rvalue } => {
                    let return_val = self.calculate_statement(&rvalue);
                    self.environment.insert_new(lvalue.clone(), return_val);
                    InternalDataStucture::Void
                }
            },
            Either::AssignmentStatement(boxed_assignment_statement) => {
                match &**boxed_assignment_statement {
                    AssignmentStatement { lvalue, rvalue } => {
                        let return_val = self.calculate_statement(&rvalue);
                        self.environment.insert(lvalue.clone(), return_val);
                        InternalDataStucture::Void
                    }
                }
            }
            Either::Identifier(name, _) => self.environment.get(&name).unwrap().clone(),

            Either::BlockStatement(statements, tipe) => {
                let mut result = InternalDataStucture::Void;
                self.environment.create_child();
                for statement in statements {
                    result = self.calculate_statement(statement);
                }
                self.environment.pop_child();
                match tipe {
                    LanguageType::Void => InternalDataStucture::Void,
                    _ => result,
                }
            }

            Either::WhileStatement(condition, block_statement, _) => {
                let mut result = InternalDataStucture::Void;
                loop {
                    match self.calculate_statement(condition) {
                        InternalDataStucture::Bool(b) if b == true => {
                            result = self.calculate_statement(block_statement);
                        }
                        InternalDataStucture::Bool(b) if b == false => break,
                        _ => unreachable!(),
                    }
                }
                result
            }
        }
    }

    fn interpret(&mut self, input: &str) {
        match self.tokenize_and_parse(input) {
            Ok((parsed_statements, new_identifiers)) => {
                for statement in parsed_statements {
                    match self.calculate_statement(&statement) {
                        InternalDataStucture::Void => continue,
                        result => println!("{:?}", result),
                    }
                }
                self.append_new_identifiers(new_identifiers);
            }
            Err(error) => println!("{:?}", error),
        }
    }

    fn interactive_prompt(&mut self) {
        'loop_start: loop {
            let mut line = String::new();
            print!(">>> ");

            let _ = std::io::stdout().flush();
            let _ = std::io::stdin().read_line(&mut line).unwrap();

            let line = line.trim_end();

            if line.len() == 0 {
                continue 'loop_start;
            }
            self.interpret(&line);
        }
    }
}
